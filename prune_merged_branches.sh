#!/bin/bash

# Function to safely prune merged branches
prune_merged_branches() {
    # Fetch latest from remote to ensure we have up-to-date information
    git fetch origin --prune

    echo "Analyzing branches..."
    
    # Get current branch
    current_branch=$(git branch --show-current)
    echo "Current branch: $current_branch"
    
    # Get list of branches whose remote tracking branch is gone
    # These are likely squash-merged branches
    gone_branches=$(git branch -vv | grep ": gone]" | awk '{print $1}')
    
    if [ -z "$gone_branches" ]; then
        echo "No branches found that were squash-merged and can be safely deleted."
        exit 0
    fi
    
    echo -e "\nThe following branches appear to be squash-merged (their remote branches are gone):"
    echo "------------------------------------------------------------------------"
    
    # Build array of candidate branches
    candidate_branches=()
    for branch in $gone_branches; do
        if [ "$branch" != "$current_branch" ] && [ "$branch" != "main" ] && [ "$branch" != "master" ]; then
            echo "$branch"
            candidate_branches+=("$branch")
        fi
    done
    
    if [ ${#candidate_branches[@]} -eq 0 ]; then
        echo "No branches found that can be safely deleted."
        exit 0
    fi
    
    echo ""
    read -p "Would you like to review branches individually? (y/n) " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Individual confirmation for each branch
        for branch in "${candidate_branches[@]}"; do
            echo ""
            echo "Branch '$branch':"
            echo "Last 3 commits:"
            git log -3 --oneline $branch
            echo ""
            read -p "Delete branch '$branch'? (y/n) " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                # Try to delete without force first
                if git branch -d $branch > /dev/null 2>&1; then
                    echo "Deleted branch: $branch"
                else
                    echo "Branch has unmerged changes. Use force delete? (y/n) "
                    read -n 1 -r
                    echo ""
                    if [[ $REPLY =~ ^[Yy]$ ]]; then
                        git branch -D $branch
                        echo "Force deleted branch: $branch"
                    else
                        echo "Skipping branch: $branch"
                    fi
                fi
            else
                echo "Skipping branch: $branch"
            fi
        done
    else
        # Bulk confirmation
        read -p "Do you want to delete all these branches? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            for branch in "${candidate_branches[@]}"; do
                if git branch -d $branch > /dev/null 2>&1; then
                    echo "Deleted branch: $branch"
                else
                    echo "Branch $branch has unmerged changes. Use force delete? (y/n) "
                    read -n 1 -r
                    echo ""
                    if [[ $REPLY =~ ^[Yy]$ ]]; then
                        git branch -D $branch
                        echo "Force deleted branch: $branch"
                    else
                        echo "Skipping branch: $branch"
                    fi
                fi
            done
        else
            echo "Operation cancelled."
        fi
    fi
}

# Run the function
prune_merged_branches